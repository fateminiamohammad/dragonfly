# @dragonfly/client

A TypeScript client for [Dragonfly](https://github.com/fateminiamohammad/dragonfly). It also works with any TypeSafe System One
API server. Answers are typed from the questions: `choice(['refund', 'delivery'] as const)` gives a
`'refund' | 'delivery'` value. It uses native `fetch` and has no runtime dependencies.

```ts
import { Dragonfly, yesNo, choice, score } from '@dragonfly/client';

const df = new Dragonfly({ baseUrl: 'http://localhost:8000', apiKey: process.env.DRAGONFLY_KEY });

const d = await df.decide(ticket, {
  urgent: yesNo('Does this need a human agent right away?'),
  intent: choice(['refund', 'delivery', 'other'] as const),
  mood: score(['angry', 'neutral', 'happy']),
});
if (d.urgent.value) route(d.intent.value, d.intent.confidence);
```

| | |
|---|---|
| `df.batch(states, questions)` | the same questions over many documents in one call (`/v1/batch`) |
| `for await (const [event, d] of df.stream(state, questions))` | `answers` (tier S, ms), `update` (tier M), `done` |
| `df.decide(state, questions, { maxError: 0.05 })` | each decision gains `decided` and `set` (guaranteed error rate) |
| `new Dragonfly({ model: 'invoices' })` or `{ model: 'auto' }` | ask a specialist, or let Dragonfly route |

Build and test from this folder: `npm install && npm run typecheck && npm test && npm run build`.
