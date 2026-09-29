import { describe, expect, it } from 'vitest';
import { choice, Dragonfly, DragonflyError, score, yesNo } from './index';

function server(handler: (path: string, body: any) => { status?: number; json?: unknown; text?: string }) {
  const calls: { path: string; body: any; headers: Record<string, string> }[] = [];
  const fetchFn = (async (url: string, init: RequestInit) => {
    const path = url.replace('http://df', '');
    const body = JSON.parse(String(init.body));
    calls.push({ path, body, headers: init.headers as Record<string, string> });
    const r = handler(path, body);
    return new Response(r.text ?? JSON.stringify(r.json), { status: r.status ?? 200 });
  }) as unknown as typeof fetch;
  return { fetchFn, calls };
}

const ANSWERS = {
  urgent: { type: 'noul', noul: 0.9, confidence: 0.8, tier: 'S' },
  intent: { type: 'choice', choice: 'refund', confidence: 0.6, probabilities: { refund: 0.8, delivery: 0.2 } },
  mood: { type: 'score', score: 0.4, confidence: 0.7, probabilities: { '0': 0.6, '1': 0.4 } },
};

describe('Dragonfly client', () => {
  it('builds typed questions and reads typed answers', async () => {
    const { fetchFn, calls } = server(() => ({ json: { answers: ANSWERS, latency_ms: 5 } }));
    const df = new Dragonfly({ baseUrl: 'http://df', apiKey: 'k', model: 'auto', fetch: fetchFn });
    const d = await df.decide('ticket', {
      urgent: yesNo('urgent?'),
      intent: choice(['refund', 'delivery'] as const),
      mood: score(['bad', 'good']),
    }, { maxError: 0.05 });
    const intent: 'refund' | 'delivery' = d.intent.value; // typed
    expect(intent).toBe('refund');
    expect(d.urgent.value).toBe(true);
    expect(d.mood.value).toBe(0.4);
    expect(d.$latencyMs).toBe(5);
    expect(calls[0].body).toEqual({
      state: 'ticket',
      model: 'auto',
      max_error: 0.05,
      questions: {
        urgent: { type: 'noul', instructions: 'urgent?' },
        intent: { type: 'choice', criteria: { refund: null, delivery: null } },
        mood: { type: 'score', criteria: ['bad', 'good'] },
      },
    });
    expect(calls[0].headers.authorization).toBe('Bearer k');
  });

  it('reads Kev-style noul answers and raises API errors', async () => {
    const { fetchFn } = server((path) =>
      path === '/v1/systemone' ? { json: { answers: { u: { type: 'noul', probability: 0.2, confidence: 0.6 } } } } : { status: 400, json: { detail: 'nope' } },
    );
    const df = new Dragonfly({ baseUrl: 'http://df', fetch: fetchFn });
    expect((await df.decide('x', { u: yesNo() })).u.value).toBe(false);
    await expect(df.batch(['a'], { u: yesNo() })).rejects.toBeInstanceOf(DragonflyError);
  });

  it('batches and streams', async () => {
    const stream = [
      `event: answers\ndata: ${JSON.stringify({ answers: { urgent: ANSWERS.urgent }, final: false })}`,
      `event: done\ndata: ${JSON.stringify({ answers: { urgent: { ...ANSWERS.urgent, tier: 'M' } }, final: true })}`,
    ].join('\n\n') + '\n\n';
    const { fetchFn } = server((path) =>
      path === '/v1/batch'
        ? { json: { results: [{ answers: { urgent: ANSWERS.urgent } }, { error: 'bad row' }] } }
        : { text: stream },
    );
    const df = new Dragonfly({ baseUrl: 'http://df', fetch: fetchFn });
    const rows = await df.batch(['a', 'b'], { urgent: yesNo() });
    expect(rows[0]).not.toBeInstanceOf(DragonflyError);
    expect(rows[1]).toBeInstanceOf(DragonflyError);
    const events = [];
    for await (const [event, d] of df.stream('x', { urgent: yesNo() })) events.push([event, d.urgent.tier]);
    expect(events).toEqual([['answers', 'S'], ['done', 'M']]);
  });
});
