/**
 * Dragonfly client: typed decisions in milliseconds. Works with any TypeSafe System One API server; Dragonfly-only
 * features (maxError, stream, batch, specialists) are optional.
 *
 *   const df = new Dragonfly({ baseUrl: 'http://localhost:8000', apiKey });
 *   const d = await df.decide(ticket, {
 *     urgent: yesNo('Does this need a human agent right away?'),
 *     intent: choice(['refund', 'delivery', 'other'] as const),
 *     mood: score(['angry', 'neutral', 'happy']),
 *   });
 *   if (d.urgent.value) route(d.intent.value);   // d.intent.value: 'refund' | 'delivery' | 'other'
 */

export interface Question<V> {
  readonly spec: Record<string, unknown>;
  /** phantom: the TypeScript type of the answer's value */
  readonly _value?: V;
}

/** A yes/no question (`noul`). Its value is a boolean. */
export function yesNo(instructions?: string): Question<boolean> {
  return { spec: { type: 'noul', ...(instructions ? { instructions } : {}) } };
}

/** One of the given options (`choice`): names, or names -> descriptions. Its value is the chosen name. */
export function choice<const K extends string>(
  options: readonly K[] | Record<K, string | null>,
  instructions?: string,
): Question<K> {
  const criteria = Array.isArray(options)
    ? Object.fromEntries((options as readonly K[]).map((o) => [o, null]))
    : options;
  return { spec: { type: 'choice', criteria, ...(instructions ? { instructions } : {}) } };
}

/** An ordered scale (`score`), lowest level first. Its value is the expected level (0 .. levels - 1). */
export function score(levels: readonly unknown[], instructions?: string): Question<number> {
  return { spec: { type: 'score', criteria: [...levels], ...(instructions ? { instructions } : {}) } };
}

export interface Decision<V> {
  type: 'noul' | 'choice' | 'score';
  value: V;
  confidence: number;
  probabilities: Record<string, number>;
  tier?: string;
  /** with maxError: the answer is within the requested error rate */
  decided?: boolean;
  /** with maxError: the options that contain the truth with probability >= 1 - maxError */
  set?: string[];
  raw: Record<string, unknown>;
}

export type Decisions<Q extends Record<string, Question<unknown>>> = {
  [K in keyof Q]: Decision<Q[K] extends Question<infer V> ? V : never>;
} & { $response: Record<string, any>; $latencyMs?: number; $specialist?: string };

export interface Options {
  baseUrl?: string;
  apiKey?: string;
  /** default model: 'dragonfly-latest', a specialist name, or 'auto' */
  model?: string;
  timeoutMs?: number;
  fetch?: typeof fetch;
}

export interface DecideOptions {
  model?: string;
  maxError?: number;
}

export class DragonflyError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(`HTTP ${status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
  }
}

function toDecision(a: Record<string, any>): Decision<any> {
  if (a.type === 'noul') {
    const p = a.noul ?? a.probability;
    return { type: 'noul', value: p >= 0.5, confidence: a.confidence, probabilities: { false: 1 - p, true: p }, tier: a.tier, decided: a.decided, set: a.set, raw: a };
  }
  const value = a.type === 'choice' ? a.choice : a.score;
  return { type: a.type, value, confidence: a.confidence, probabilities: a.probabilities ?? {}, tier: a.tier, decided: a.decided, set: a.set, raw: a };
}

function toDecisions<Q extends Record<string, Question<unknown>>>(body: Record<string, any>): Decisions<Q> {
  const out: Record<string, unknown> = {};
  for (const [k, a] of Object.entries(body.answers ?? {})) out[k] = toDecision(a as Record<string, any>);
  return Object.assign(out, { $response: body, $latencyMs: body.latency_ms, $specialist: body.specialist }) as Decisions<Q>;
}

export class Dragonfly {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly fetchFn: typeof fetch;

  constructor(private readonly options: Options = {}) {
    this.baseUrl = (options.baseUrl ?? 'http://localhost:8000').replace(/\/$/, '');
    this.headers = { 'content-type': 'application/json' };
    if (options.apiKey) this.headers.authorization = `Bearer ${options.apiKey}`;
    this.fetchFn = options.fetch ?? fetch;
  }

  private body(state: unknown, questions: Record<string, Question<unknown>>, opts: DecideOptions) {
    const body: Record<string, unknown> = {
      state,
      questions: Object.fromEntries(Object.entries(questions).map(([k, q]) => [k, q.spec])),
    };
    const model = opts.model ?? this.options.model;
    if (model) body.model = model;
    if (opts.maxError !== undefined) body.max_error = opts.maxError;
    return body;
  }

  private async post(path: string, body: unknown): Promise<Response> {
    const res = await this.fetchFn(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.options.timeoutMs ?? 30_000),
    });
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail ?? j, () => res.statusText);
      throw new DragonflyError(res.status, detail);
    }
    return res;
  }

  async decide<Q extends Record<string, Question<unknown>>>(state: unknown, questions: Q, opts: DecideOptions = {}): Promise<Decisions<Q>> {
    const res = await this.post('/v1/systemone', this.body(state, questions, opts));
    return toDecisions<Q>(await res.json());
  }

  /** The same questions over many states in one call (/v1/batch). A failed row is a DragonflyError in its place. */
  async batch<Q extends Record<string, Question<unknown>>>(
    states: unknown[],
    questions: Q,
    opts: DecideOptions = {},
  ): Promise<(Decisions<Q> | DragonflyError)[]> {
    const res = await this.post('/v1/batch', { requests: states.map((s) => this.body(s, questions, opts)) });
    const out = await res.json();
    return out.results.map((r: Record<string, any>) => ('error' in r ? new DragonflyError(422, r.error) : toDecisions<Q>(r)));
  }

  /** Anytime answers: 'answers' (tier S, ms), maybe 'update' (tier M's upgrades), then 'done' (the final body). */
  async *stream<Q extends Record<string, Question<unknown>>>(
    state: unknown,
    questions: Q,
    opts: DecideOptions = {},
  ): AsyncGenerator<[event: 'answers' | 'update' | 'done', decisions: Decisions<Q>]> {
    const res = await this.post('/v1/systemone?stream=true', this.body(state, questions, opts));
    const text = await res.text();
    for (const block of text.split('\n\n')) {
      const lines = block.split('\n');
      const event = lines.find((l) => l.startsWith('event: '))?.slice(7);
      const data = lines.filter((l) => l.startsWith('data: ')).map((l) => l.slice(6)).join('');
      if (!event || !data) continue;
      const parsed = JSON.parse(data);
      if (event === 'error') throw new DragonflyError(parsed.status ?? 500, parsed.detail);
      yield [event as 'answers' | 'update' | 'done', toDecisions<Q>(parsed)];
    }
  }
}
