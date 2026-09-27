/** Thin client for the backend (/api). The JWT lives in localStorage so a reload keeps you logged in. */

const TOKEN = 'dragonfly.token';

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN, token);
    else localStorage.removeItem(TOKEN);
  } catch {
    /* storage blocked: the session just won't survive a reload */
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  const token = getToken();
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api${path}`, { ...init, headers });
  const body = await res.json().catch(() => ({}));
  if (res.status === 401 && path !== '/auth/login') {
    setToken(null);
    window.dispatchEvent(new Event('dragonfly:logout'));
  }
  if (!res.ok) {
    const detail = body?.message ?? body?.detail ?? res.statusText;
    throw new ApiError(Array.isArray(detail) ? detail.join(', ') : String(detail), res.status);
  }
  return body as T;
}

export interface Answer {
  type: 'choice' | 'noul' | 'score';
  choice?: string;
  noul?: number;
  score?: number;
  confidence: number;
  probabilities?: Record<string, number>;
  legend?: Record<string, string>;
  tier?: string;
}

export interface DecideResponse {
  answers: Record<string, Answer>;
  latency_ms: number;
  cached?: boolean;
  usage: { input_tokens: number };
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  revokedAt: string | null;
  key?: string;
}

export interface UsageRow {
  day: string;
  keyId: string;
  keyName: string;
  requests: number;
  questions: number;
  tokens: number;
}
