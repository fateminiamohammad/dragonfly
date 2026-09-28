import { useEffect, useState } from 'react';
import { api, getToken, setToken } from './api';
import { Keys } from './pages/Keys';
import { Login } from './pages/Login';
import { Media } from './pages/Media';
import { Model } from './pages/Model';
import { Playground } from './pages/Playground';
import { Review } from './pages/Review';
import { Specialists } from './pages/Specialists';
import { Usage } from './pages/Usage';

const PAGES = {
  playground: 'Playground',
  media: 'Media',
  specialists: 'Specialists',
  review: 'Review',
  keys: 'API keys',
  usage: 'Usage',
  model: 'Model & plugins',
} as const;
type Page = keyof typeof PAGES;

function pageFromHash(): Page {
  const h = window.location.hash.slice(1);
  return h in PAGES ? (h as Page) : 'playground';
}

export function App() {
  const [email, setEmail] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    const onLogout = () => setEmail(null);
    window.addEventListener('hashchange', onHash);
    window.addEventListener('dragonfly:logout', onLogout);
    if (getToken()) {
      api<{ email: string }>('/me')
        .then((u) => setEmail(u.email))
        .catch(() => setEmail(null))
        .finally(() => setChecked(true));
    } else setChecked(true);
    return () => {
      window.removeEventListener('hashchange', onHash);
      window.removeEventListener('dragonfly:logout', onLogout);
    };
  }, []);

  if (!checked) return null;
  if (!email) return <Login onLogin={setEmail} />;

  return (
    <div className="shell">
      <header>
        <span className="brand">
          <span className="logo">◇</span> Dragonfly
        </span>
        <nav>
          {(Object.keys(PAGES) as Page[]).map((p) => (
            <a key={p} href={`#${p}`} className={p === page ? 'active' : ''}>
              {PAGES[p]}
            </a>
          ))}
        </nav>
        <span className="grow" />
        <span className="muted small">{email}</span>
        <button
          onClick={() => {
            setToken(null);
            setEmail(null);
          }}
        >
          Sign out
        </button>
      </header>
      <main>
        {page === 'playground' && <Playground />}
        {page === 'media' && <Media />}
        {page === 'specialists' && <Specialists />}
        {page === 'review' && <Review />}
        {page === 'keys' && <Keys />}
        {page === 'usage' && <Usage />}
        {page === 'model' && <Model />}
      </main>
    </div>
  );
}
